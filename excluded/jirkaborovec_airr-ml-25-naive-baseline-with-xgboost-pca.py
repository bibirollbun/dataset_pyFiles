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


# Iterate over all sub-datasets
dataset = []
for folder in tqdm(train_datasets):
    path_dataset_ = os.path.join(PATH_TRAIN_DATASETS, folder)
    dataset += parse_tsv_files(path_dataset_)

dataset_train = pd.DataFrame(dataset)
display(dataset_train)


# Iterate over all sub-datasets
dataset = []
for folder in tqdm(test_datasets):
    path_dataset_ = os.path.join(PATH_TEST_DATASETS, folder)
    dataset += parse_tsv_files(path_dataset_)

dataset_test = pd.DataFrame(dataset)
display(dataset_test)


import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.decomposition import PCA # Import PCA

# Prepare the data
X_train = dataset_train.drop(
    ['label_positive', 'ID', 'dataset'], axis=1, errors='ignore').fillna(0)
y_train = dataset_train['label_positive']

X_test = dataset_test.drop(
    ['label_positive', 'ID', 'dataset'], axis=1, errors='ignore').fillna(0)
y_test = pd.Series([]) # Initialize y_test as an empty Series as label_positive is not in test data

# Align columns - crucial for XGBoost to avoid 'feature_names mismatch'
# Ensure X_test only has columns present in X_train
missing_in_test_but_in_train = set(X_train.columns) - set(X_test.columns)
for c in missing_in_test_but_in_train:
    X_test[c] = 0

X_test = X_test[X_train.columns] # Crop columns in X_test that are not in X_train and ensure order

print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")

# Apply PCA
pca = PCA(n_components=0.95) # Retain 95% of variance
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

print(f"X_train_pca shape: {X_train_pca.shape}")
print(f"X_test_pca shape: {X_test_pca.shape}")

# Initialize XGBoost Classifier
model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', use_label_encoder=False, random_state=42)

# Train the model with PCA-transformed data
print("\nTraining XGBoost model...")
model.fit(X_train_pca, y_train)
print("Model training complete.")

# Make predictions on the test set with PCA-transformed data
print("\nMaking predictions on the test set...")
y_pred_proba = model.predict_proba(X_test_pca)[:, 1] # Get probabilities for the positive class
predictions_df = pd.DataFrame({
    'ID': dataset_test['ID'],
    'dataset': dataset_test['dataset'],
    'label_positive_pred': y_pred_proba, # Store probabilities here
})
print(f"Prediction has {len(predictions_df)} rows")
display(predictions_df.head())


import pandas as pd
import os

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
    on=['ID'],
    how='left',
)

# Rename 'label_positive_pred' to 'label_positive' for the final submission format
submission_df = submission_df.rename(columns={'label_positive_pred': 'label_positive_probability'})
submission_df = submission_df.fillna(0.5)
# Remove the duplicates from the final result, keeping the first occurrence
submission_df = submission_df.drop_duplicates(subset=['ID'], keep='first')

print(f"Sample submissions has {len(submission_df)} rows")
# Display the head of the final submission DataFrame
print("Final Submission DataFrame Head:")
display(submission_df.head())

# You can save this to a CSV file if needed
submission_df.to_csv('submission.csv', index=False)


!head 'submission.csv'

