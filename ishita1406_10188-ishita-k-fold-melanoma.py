import numpy as np
import pandas as pd
from sklearn import model_selection
import os
from pathlib import Path

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("Libraries imported successfully!")


DATA_PATH = "/kaggle/input/siim-isic-melanoma-classification/train.csv"
NUM_FOLDS_5 = 5
NUM_FOLDS_10 = 10
TARGET_COL = "target"
RANDOM_STATE = 42

print("Configuration set up!")


def create_stratified_folds(data, target_col='target', num_splits=5, random_state=RANDOM_STATE):
    """
    Create stratified k-fold splits for regression/classification problems.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input dataframe containing features and target
    target_col : str
        Name of the target column
    num_splits : int
        Number of folds to create
    random_state : int
        Random state for reproducibility
        
    Returns:
    --------
    pandas.DataFrame
        Dataframe with added 'kfold' column
    """
    
    data = data.copy()
    data["kfold"] = -1
    
    num_bins = int(np.floor(1 + np.log2(len(data))))
    print(f'Number of bins created: {num_bins}')
    
    data.loc[:, "bins"] = pd.cut(
        data[target_col], 
        bins=num_bins, 
        labels=False,
        duplicates='drop'
    )
    
    kf = model_selection.StratifiedKFold(
        n_splits=num_splits, 
        shuffle=True, 
        random_state=random_state
    )
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X=data, y=data.bins.values)):
        data.loc[val_idx, 'kfold'] = fold
    
    data = data.drop("bins", axis=1)
    
    return data


print("Loading dataset...")
df = pd.read_csv(DATA_PATH)

print(f"Dataset shape: {df.shape}")
print(f"Target distribution:\n{df[TARGET_COL].value_counts()}")
print(f"Target value counts:\n{df[TARGET_COL].value_counts(normalize=True)}")

print("\nDataset info:")
print(df.info())
print("\nFirst few rows:")
df.head()


print("Creating 5-fold splits...")
df_5_folds = create_stratified_folds(df, target_col=TARGET_COL, num_splits=NUM_FOLDS_5)

print("5-fold splits created successfully!")
print(f"Fold distribution:\n{df_5_folds['kfold'].value_counts().sort_index()}")

print("\nTarget distribution per fold (5-folds):")
for fold in range(NUM_FOLDS_5):
    fold_data = df_5_folds[df_5_folds['kfold'] == fold]
    print(f"Fold {fold}: {len(fold_data)} samples, "
          f"Target mean: {fold_data[TARGET_COL].mean():.4f}, "
          f"Positive samples: {fold_data[TARGET_COL].sum()}")

df_5_folds.head()


print("Creating 10-fold splits...")
df_10_folds = create_stratified_folds(df, target_col=TARGET_COL, num_splits=NUM_FOLDS_10)


print("10-fold splits created successfully!")
print(f"Fold distribution:\n{df_10_folds['kfold'].value_counts().sort_index()}")


print("\nTarget distribution per fold (10-folds):")
for fold in range(NUM_FOLDS_10):
    fold_data = df_10_folds[df_10_folds['kfold'] == fold]
    print(f"Fold {fold}: {len(fold_data)} samples, "
          f"Target mean: {fold_data[TARGET_COL].mean():.4f}, "
          f"Positive samples: {fold_data[TARGET_COL].sum()}")

df_10_folds.head()


output_5_folds = "train_5folds.csv"
output_10_folds = "train_10folds.csv"

df_5_folds.to_csv(output_5_folds, index=False)
df_10_folds.to_csv(output_10_folds, index=False)

print("Files saved successfully!")
print(f"5-fold data saved as: {output_5_folds}")
print(f"10-fold data saved as: {output_10_folds}")

# Verify file creation
print(f"\nFile sizes:")
print(f"5-folds file: {os.path.getsize(output_5_folds)} bytes")
print(f"10-folds file: {os.path.getsize(output_10_folds)} bytes")


print("=== SUMMARY STATISTICS ===")
print(f"Original dataset size: {len(df)}")
print(f"Original target mean: {df[TARGET_COL].mean():.4f}")
print(f"Original positive samples: {df[TARGET_COL].sum()}")

print(f"\n5-Fold Configuration:")
print(f"Average samples per fold: {len(df_5_folds) / NUM_FOLDS_5:.1f}")
print(f"Fold sizes: {df_5_folds['kfold'].value_counts().sort_index().tolist()}")

print(f"\n10-Fold Configuration:")
print(f"Average samples per fold: {len(df_10_folds) / NUM_FOLDS_10:.1f}")
print(f"Fold sizes: {df_10_folds['kfold'].value_counts().sort_index().tolist()}")


target_mean_5folds = [df_5_folds[df_5_folds['kfold'] == fold][TARGET_COL].mean() for fold in range(NUM_FOLDS_5)]
target_mean_10folds = [df_10_folds[df_10_folds['kfold'] == fold][TARGET_COL].mean() for fold in range(NUM_FOLDS_10)]

print(f"\nTarget mean across 5-folds: {target_mean_5folds}")
print(f"Target mean std (5-folds): {np.std(target_mean_5folds):.6f}")

print(f"\nTarget mean across 10-folds: {[f'{x:.4f}' for x in target_mean_10folds]}")
print(f"Target mean std (10-folds): {np.std(target_mean_10folds):.6f}")



print("=== VERIFICATION ===")
print(f"5-folds - Samples without fold assignment: {(df_5_folds['kfold'] == -1).sum()}")
print(f"10-folds - Samples without fold assignment: {(df_10_folds['kfold'] == -1).sum()}")

print(f"5-folds - Unique fold values: {sorted(df_5_folds['kfold'].unique())}")
print(f"10-folds - Unique fold values: {sorted(df_10_folds['kfold'].unique())}")

print(f"\nOriginal samples preserved in 5-folds: {len(df_5_folds) == len(df)}")
print(f"Original samples preserved in 10-folds: {len(df_10_folds) == len(df)}")

print("\nFold creation completed successfully!")


print("5-Fold Dataset Sample:")
display(df_5_folds.head(10))

print("\n10-Fold Dataset Sample:")
display(df_10_folds.head(10))

