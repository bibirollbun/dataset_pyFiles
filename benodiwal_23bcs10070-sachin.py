import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import os

# Configuration
N_FOLDS = 5
RANDOM_STATE = 42
N_BINS = 10  # For stratification

# Paths
INPUT_DIR = '/kaggle/input/petfinder-pawpularity-score'
OUTPUT_DIR = '/kaggle/working'

def create_bins(df, n_bins=10):
    """
    Create bins for stratified sampling based on target distribution
    Uses Sturges' rule for optimal binning
    """
    df['bins'] = pd.cut(df['Pawpularity'], bins=n_bins, labels=False)
    return df

def create_folds(df, n_splits=5, random_state=42):
    """
    Create stratified K-Fold splits based on binned target values
    """
    df['fold'] = -1
    
    # Initialize StratifiedKFold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    # Create folds
    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['bins'])):
        df.loc[val_idx, 'fold'] = fold
    
    return df

def main():
    print("=" * 60)
    print("PetFinder Pawpularity - K-Fold Creation")
    print("=" * 60)
    
    # Load training data
    print(f"\nLoading training data from {INPUT_DIR}...")
    train_df = pd.read_csv(f'{INPUT_DIR}/train.csv')
    
    print(f"Total samples: {len(train_df)}")
    print(f"\nTarget distribution:")
    print(train_df['Pawpularity'].describe())
    
    # Create bins for stratification
    print(f"\nCreating {N_BINS} bins for stratified sampling...")
    train_df = create_bins(train_df, n_bins=N_BINS)
    
    # Check bin distribution
    print("\nBin distribution:")
    print(train_df['bins'].value_counts().sort_index())
    
    # Create folds
    print(f"\nCreating {N_FOLDS} stratified folds...")
    train_df = create_folds(train_df, n_splits=N_FOLDS, random_state=RANDOM_STATE)
    
    # Verify fold distribution
    print("\nFold distribution:")
    for fold in range(N_FOLDS):
        fold_df = train_df[train_df['fold'] == fold]
        print(f"Fold {fold}: {len(fold_df)} samples, "
              f"Mean Pawpularity: {fold_df['Pawpularity'].mean():.2f}, "
              f"Std: {fold_df['Pawpularity'].std():.2f}")
    
    # Drop the bins column (not needed for training)
    train_df = train_df.drop('bins', axis=1)
    
    # Save the dataframe with fold information
    output_path = f'{OUTPUT_DIR}/train_folds.csv'
    train_df.to_csv(output_path, index=False)
    print(f"\nSaved fold information to: {output_path}")
    
    # Verify saved file
    print("\nFirst few rows of saved data:")
    print(train_df.head())
    
    print("\n" + "=" * 60)
    print("K-Fold creation completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()

