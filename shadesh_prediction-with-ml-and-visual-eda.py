import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data with enhanced debugging
base_path = "/kaggle/input/stanford-rna-3d-folding"

def debug_load(file_name):
    path = f"{base_path}/{file_name}"
    print(f"\nLoading {file_name}...")
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return None
    
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} rows")
    print("Sample data:")
    display(df.head(2))
    print("Columns:", df.columns.tolist())
    return df

# Load all files with debugging
print("===== Loading Data Files =====")
train_seq = debug_load("train_sequences.csv")
train_labels = debug_load("train_labels.csv")
val_seq = debug_load("validation_sequences.csv")
val_labels = debug_load("validation_labels.csv")
test_seq = debug_load("test_sequences.csv")

# Verify merge keys exist
print("\n===== Checking Merge Keys =====")
for df, name in [(train_seq, "train_seq"), (train_labels, "train_labels"),
                 (val_seq, "val_seq"), (val_labels, "val_labels")]:
    if df is not None:
        print(f"{name} has 'target_id' column: {'target_id' in df.columns}")

# Standardize column names
if train_labels is not None and 'ID' in train_labels.columns:
    train_labels.rename(columns={'ID': 'target_id'}, inplace=True)
if val_labels is not None and 'ID' in val_labels.columns:
    val_labels.rename(columns={'ID': 'target_id'}, inplace=True)

# Merge with intersection to ensure valid matches
print("\n===== Merging Data =====")
def safe_merge(left, right, name):
    if left is None or right is None:
        print(f"Cannot merge {name} - one or both DataFrames are None")
        return None
    
    common_ids = set(left['target_id']).intersection(set(right['target_id']))
    print(f"{name}: {len(common_ids)} common target_ids")
    
    merged = pd.merge(left, right, on='target_id', how='inner')
    print(f"Merged {len(merged)} rows (from {len(left)} + {len(right)})")
    if len(merged) > 0:
        print("Merged columns:", merged.columns.tolist())
    return merged

train_data = safe_merge(train_seq, train_labels, "Training Data")
val_data = safe_merge(val_seq, val_labels, "Validation Data")

# Basic EDA only if we have data
if train_data is not None and len(train_data) > 0:
    print("\n===== Available Columns for EDA =====")
    print(train_data.columns.tolist())
    
    # Sequence length distribution
    if 'sequence' in train_data.columns:
        plt.figure(figsize=(12, 6))
        train_data['sequence'].str.len().hist(bins=30)
        plt.title('Sequence Length Distribution')
        plt.xlabel('Length')
        plt.ylabel('Count')
        plt.show()
    
    # Residue distribution if column exists
    if 'resname' in train_data.columns:
        plt.figure(figsize=(12, 6))
        sns.countplot(x=train_data['resname'].dropna())
        plt.title('Residue Distribution')
        plt.xticks(rotation=45)
        plt.show()
    else:
        print("'resname' column not found for distribution plot")
else:
    print("\nNo training data available for EDA")

# Only proceed with modeling if we have valid data
if (train_data is not None and len(train_data) > 0 and 
    test_seq is not None and len(test_seq) > 0):
    
    print("\n===== Preparing Baseline Model =====")
    # Prepare features (using sequence length)
    train_data['seq_length'] = train_data['sequence'].str.len()
    test_seq['seq_length'] = test_seq['sequence'].str.len()
    
    # Train simple model
    from sklearn.neighbors import KNeighborsRegressor
    knn = KNeighborsRegressor(n_neighbors=3)
    knn.fit(train_data[['seq_length']], train_data[['x_1', 'y_1', 'z_1']])
    
    # Make predictions
    preds = knn.predict(test_seq[['seq_length']])
    
    # Create submission
    submission = pd.DataFrame({
        'ID': test_seq['target_id'] + "_centroid",
        'x_1': preds[:, 0],
        'y_1': preds[:, 1],
        'z_1': preds[:, 2]
    })
    submission.to_csv("/kaggle/working/submission.csv", index=False)
    print("Submission saved:")
    display(submission.head())
else:
    print("\nInsufficient data for modeling")




